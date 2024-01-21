DROP DATABASE IF EXISTS inz;
CREATE DATABASE inz;
USE inz;
DROP SCHEMA IF EXISTS heart_disease;
CREATE SCHEMA heart_disease;


CREATE TABLE inz.attribute (
        id integer not null unique,
		attribute_name varchar(25),
		attribute_field_name varchar(25),
	primary key (id)
    ) ENGINE=InnoDB;

CREATE TABLE inz.categorised_data (
        id integer not null unique,
		A1 varchar(12),
		A2 varchar(8),
		A3 varchar(7),
		A4 varchar(7),
		A5 varchar(7),
		class varchar(8),
	primary key (id)
    ) ENGINE=InnoDB;

INSERT INTO inz.attribute (id, attribute_name, attribute_field_name) VALUES
(1, 'What is age?', 'A1'),
(2, 'What is sex?', 'A2'),
(3, 'What is pulse?', 'A3'),
(4, 'What is SAP?', 'A4'),
(5, 'What is DAP?', 'A5');

	
INSERT INTO inz.categorised_data (id, A1, A2, A3, A4, A5, class) VALUES 
(1,'senior','female','normal','high','high','ill'),
(2,'senior','female','normal','high','normal','healthy'),
(3,'middle_aged','male','high','high','normal','ill'),
(4,'senior','female','high','high','high','ill'),
(5,'senior','female','normal','high','high','healthy'),
(6,'senior','female','normal','high','normal','ill'),
(7,'senior','female','normal','high','high','ill'),
(8,'senior','female','normal','high','normal','ill'),
(9,'middle_aged','female','normal','high','high','healthy'),
(10,'senior','male','low','high','normal','healthy'),
(11,'senior','female','normal','high','high','healthy'),
(12,'senior','female','normal','high','high','ill'),
(13,'middle_aged','male','high','high','normal','healthy'),
(14,'middle_aged','female','normal','high','normal','healthy'),
(15,'middle_aged','female','normal','high','normal','healthy'),
(16,'middle_aged','female','normal','high','normal','healthy'),
(17,'middle_aged','female','low','high','high','healthy'),
(18,'senior','female','normal','high','high','healthy'),
(19,'youth','female','normal','high','normal','healthy'),
(20,'middle_aged','female','normal','high','normal','ill'),
(21,'middle_aged','male','normal','high','normal','healthy'),
(22,'senior','female','normal','normal','normal','ill'),
(23,'middle_aged','female','normal','high','normal','healthy'),
(24,'senior','female','high','high','high','healthy'),
(25,'senior','female','normal','high','normal','healthy'),
(26,'middle_aged','female','normal','high','high','healthy'),
(27,'senior','female','normal','high','normal','healthy'),
(28,'senior','male','normal','high','normal','healthy'),
(29,'middle_aged','female','normal','high','normal','healthy'),
(30,'middle_aged','male','normal','high','normal','healthy'),
(31,'senior','female','high','high','normal','ill'),
(32,'senior','female','high','high','high','healthy'),
(33,'senior','male','normal','high','normal','healthy'),
(34,'senior','female','normal','high','normal','healthy'),
(35,'senior','male','high','high','normal','healthy'),
(36,'senior','female','high','high','high','healthy'),
(37,'senior','female','high','high','high','ill'),
(38,'senior','female','normal','high','high','healthy'),
(39,'middle_aged','male','normal','high','normal','healthy'),
(40,'middle_aged','female','normal','high','normal','healthy'),
(41,'senior','male','normal','high','normal','healthy'),
(42,'middle_aged','male','high','high','normal','healthy'),
(43,'senior','male','high','high','normal','healthy'),
(44,'senior','female','normal','normal','normal','healthy'),
(45,'senior','female','normal','high','normal','healthy'),
(46,'senior','female','normal','high','normal','healthy'),
(47,'senior','female','high','high','high','ill'),
(48,'middle_aged','female','normal','high','normal','healthy'),
(49,'senior','female','normal','high','high','healthy'),
(50,'senior','female','normal','high','high','healthy'),
(51,'senior','male','low','high','high','healthy'),
(52,'senior','male','normal','high','normal','healthy'),
(53,'senior','female','normal','high','normal','healthy'),
(54,'middle_aged','male','normal','high','normal','healthy'),
(55,'senior','female','normal','high','high','healthy'),
(56,'senior','female','high','normal','high','healthy'),
(57,'middle_aged','female','low','normal','normal','healthy'),
(58,'middle_aged','female','normal','high','normal','healthy'),
(59,'senior','female','normal','high','normal','ill'),
(60,'senior','female','low','high','high','healthy'),
(61,'senior','female','normal','high','normal','healthy'),
(62,'senior','female','normal','high','high','healthy'),
(63,'middle_aged','female','normal','high','high','healthy')